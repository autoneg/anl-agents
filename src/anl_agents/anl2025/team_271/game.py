from anl2025.ufun import CenterUFun
from typing import Iterable
from negmas.outcomes import Outcome

class GameEnvironment(): 
    def __init__(self, center_ufun: CenterUFun, n_edges: int, outcome_space_at_subnegotiation: list[list[Outcome]]):
        self.center_ufun = center_ufun
        self.n_edges = n_edges
        self.outcome_space_at_subnegotiation = outcome_space_at_subnegotiation

class AbstractedOutcomeGameState():
    def __init__(self, history: tuple[Outcome], environment: GameEnvironment):
        self.history = history 
        self._environment = environment
    
    def __str__(self):
        return str(self.history)

    def __hash__(self):
        return hash(self.history)

    def __eq__(self, other):
        return isinstance(other, AbstractedOutcomeGameState) and self.history == other.history
    
    def get_actions(self) -> Iterable[Outcome]:
        # Get all possible actions from the center utility function
        current_index = self.get_current_negotiation_index()
        outcomes = self._environment.outcome_space_at_subnegotiation[current_index]
        return outcomes
    
    def get_child_from_action(self, action: Outcome) -> "AbstractedOutcomeGameState":
        return AbstractedOutcomeGameState(self.history + (action,), self._environment)

    def get_children(self) -> list["AbstractedOutcomeGameState"]:
        
        if self.is_terminal():
            return []
        
        children = []

        # Find every possible outcome in the outcome space for the subnegotiation
        outcomes = self.get_actions()

        for outcome in outcomes:
            # Create a new game state for each action
            child = AbstractedOutcomeGameState(self.history + (outcome,), self._environment)
            children.append(child)
        
        return children

    def get_current_negotiation_index(self) -> int:
        return len(self.history)

    def is_terminal(self) -> bool: 
        if self.get_current_negotiation_index() == self._environment.n_edges:
            return True
        else:
            return False
        
    def get_current_utility(self) -> float:
        if self.history == (): 
            return self._environment.center_ufun.reserved_value
        # Get the utility of the current state
        current_outcome = [None for _ in range(self._environment.n_edges)]
        any_agreement = False
        for index, agreement in enumerate(self.history):
            if agreement is not None:
                current_outcome[index] = agreement
                any_agreement = True
        utility = self._environment.center_ufun.eval(current_outcome)

        if any_agreement:
            return utility
        else:
            return self._environment.center_ufun.reserved_value  
