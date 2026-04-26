from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import random


def _default_float_dict():
    """Helper function to create defaultdict(float) - replaces lambda for pickling."""
    return defaultdict(float)


class SimpleFrequencyOpponentModel:
    """
    Simple frequency-based opponent model.
    
    Two key heuristics:
    1. The opponent concedes less on important issues
    2. Preferred values appear more often in opponent's offers
    """
    
    def __init__(self, outcome_space: List[Tuple], alpha: float = 0.1):
        self.alpha = alpha
        self.opponent_bids = []
        
        # Filter out None values from outcome space
        valid_outcome_space = [outcome for outcome in outcome_space if outcome is not None]
        self.outcome_space = valid_outcome_space
        
        # Get number of issues from outcome space
        self.num_issues = len(valid_outcome_space[0]) if valid_outcome_space else 0
        
        # Issue importance weights: {issue_index: weight}
        self.issue_weights = {}
        
        # Value frequencies: {issue_index: {value: count}}
        self.value_counts = defaultdict(_default_float_dict)
        
        # Initialize with uniform weights and small counts
        for i in range(self.num_issues):
            self.issue_weights[i] = 1.0 / self.num_issues
            
        # Initialize value counts - only with valid outcomes
        for outcome in valid_outcome_space:
            for issue_idx, value in enumerate(outcome):
                self.value_counts[issue_idx][value] += 0.1
    
    def update(self, bid: Tuple, t: float):
        """Update the model with a new opponent bid."""
        self.opponent_bids.append(bid)
        
        # Update value frequencies
        for issue_idx, value in enumerate(bid):
            self.value_counts[issue_idx][value] += 1.0
        
        # Update issue importance (if opponent didn't change value, issue is important)
        if len(self.opponent_bids) >= 2:
            previous_bid = self.opponent_bids[-2]
            for issue_idx in range(min(len(bid), len(previous_bid))):
                if bid[issue_idx] == previous_bid[issue_idx]:
                    self.issue_weights[issue_idx] += self.alpha * (1.0 - t)
        
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize issue weights to sum to 1."""
        total = sum(self.issue_weights.values())
        if total > 0:
            for issue_idx in self.issue_weights:
                self.issue_weights[issue_idx] /= total
    
    def predict_utility(self, outcome: Tuple) -> float:
        """Predict opponent utility for an outcome."""
        if not self.opponent_bids:
            return 0.5
        
        total_utility = 0.0
        
        for issue_idx, value in enumerate(outcome):
            if issue_idx in self.value_counts and value in self.value_counts[issue_idx]:
                # Get normalized value utility (frequency / max_frequency)
                value_count = self.value_counts[issue_idx][value]
                max_count = max(self.value_counts[issue_idx].values()) if self.value_counts[issue_idx] else 1.0
                value_utility = value_count / max_count if max_count > 0 else 0.0
                
                # Weight by issue importance
                issue_weight = self.issue_weights.get(issue_idx, 1.0 / self.num_issues)
                total_utility += value_utility * issue_weight
        
        return max(0.0, min(1.0, total_utility))
    
    def calculate_nash_product(self, outcome: Tuple, my_utility: float) -> float:
        """
        Calculate Nash product (my_utility × opponent_utility).
        
        Args:
            outcome: The outcome to evaluate
            my_utility: My utility for this outcome
            
        Returns:
            Nash product value
        """
        opponent_utility = self.predict_utility(outcome)
        return my_utility * opponent_utility
    
    def is_likely_acceptable(self, outcome: Tuple, negotiation_time: float) -> bool:
        """
        Predict if an outcome would be acceptable to the opponent.
        
        Args:
            outcome: The outcome to check
            negotiation_time: Current negotiation time
            
        Returns:
            True if likely acceptable, False otherwise
        """
        predicted_utility = self.predict_utility(outcome)
        
        # Get opponent's estimated reservation value (minimum utility they've shown)
        if self.opponent_bids:
            min_observed_utility = min(self.predict_utility(bid) for bid in self.opponent_bids)
            reservation_estimate = min_observed_utility * 0.9  # Conservative estimate
        else:
            reservation_estimate = 0.3  # Default assumption
        
        # Account for time pressure - opponents become less demanding over time
        time_adjusted_threshold = reservation_estimate * (1.0 - negotiation_time * 0.3)
        
        return predicted_utility >= time_adjusted_threshold
    
    def get_best_bid_by_nash(self, candidate_bids: List[Tuple], my_utility_func) -> Optional[Tuple]:
        """
        Select the bid with the highest Nash product from candidates.
        
        Args:
            candidate_bids: List of candidate bids to choose from
            my_utility_func: Function to calculate my utility for a bid
            
        Returns:
            Best bid according to Nash product, or None if no candidates
        """
        if not candidate_bids:
            return None
        
        best_bid = None
        best_nash_product = -1.0
        
        for bid in candidate_bids:
            my_util = my_utility_func(bid)
            nash_product = self.calculate_nash_product(bid, my_util)
            
            if nash_product > best_nash_product:
                best_nash_product = nash_product
                best_bid = bid
        
        return best_bid
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the current model state."""
        return {
            "total_bids_observed": len(self.opponent_bids),
            "issue_weights": dict(self.issue_weights),
            "value_frequencies": {
                issue: dict(values) for issue, values in self.value_counts.items()
            },
            "most_recent_bids": self.opponent_bids[-3:] if len(self.opponent_bids) >= 3 else self.opponent_bids,
        }


# Test
if __name__ == "__main__":
    outcome_space = [
        ("High", "Premium", "Fast"),
        ("High", "Standard", "Normal"),
        ("Medium", "Premium", "Fast"),
        ("Low", "Basic", "Slow"),
    ]
    
    model = SimpleFrequencyOpponentModel(outcome_space)
    
    # Simulate opponent bids
    bids = [
        ("High", "Premium", "Fast"),
        ("High", "Premium", "Normal"),
        ("High", "Standard", "Fast"),
    ]
    
    for i, bid in enumerate(bids):
        t = i / len(bids)
        model.update(bid, t)
        utility = model.predict_utility(bid)
        pass # print(f"Bid: {bid}, Predicted utility: {utility:.3f}")
    
    pass # print(f"Issue weights: {model.issue_weights}")
    pass # print(f"Value counts: {dict(model.value_counts)}")
    
    # Test Nash product calculation
    def my_utility(outcome):
        # Dummy utility function - prefer Low quality, Basic service, Slow delivery
        quality_util = {"Low": 1.0, "Medium": 0.5, "High": 0.0}[outcome[0]]
        service_util = {"Basic": 1.0, "Standard": 0.5, "Premium": 0.0}[outcome[1]]
        speed_util = {"Slow": 1.0, "Normal": 0.5, "Fast": 0.0}[outcome[2]]
        return (quality_util + service_util + speed_util) / 3.0
    
    # Test bid selection
    candidates = [
        ("Medium", "Standard", "Normal"),
        ("High", "Premium", "Normal"),
        ("Low", "Premium", "Fast"),
    ]
    
    best_bid = model.get_best_bid_by_nash(candidates, my_utility)
    pass # print(f"\nBest bid by Nash product: {best_bid}")
    
    for bid in candidates:
        my_util = my_utility(bid)
        nash_product = model.calculate_nash_product(bid, my_util)
        pass # print(f"  {bid}: my_util={my_util:.3f}, nash_product={nash_product:.3f}")
