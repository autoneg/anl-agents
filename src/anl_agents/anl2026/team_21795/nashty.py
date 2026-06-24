import random
from negmas.sao import ResponseType
from negmas.sao.components.offering import TimeBasedOfferingPolicy
from negmas.sao.components.acceptance import ACNext
from negmas.gb.components.genius.models import GSmithFrequencyModel
from negmas.sao.negotiators.modular import BOANegotiator

class BiddingStrategy(TimeBasedOfferingPolicy):
    """
    Bidding Mode Options (randomize_offers):
    - "greedy": Always offers the absolute best deal.
    - "randomize": Randomly swaps between top 10% utility offers.
    - "feint": Offers the deal closest to but strictly below the midpoint 
               between Max Utility and Reservation Value until 90% of the game.
    
    Concession Mode (concession):
    - If True: At 90% of the game, drops utility threshold in strict 10% buckets, 
               offering the deal that maximizes opponent utility within that bucket.
               If a bucket is empty, falls back to the previous offer.
    """
    def __init__(self, utility_threshold_ratio: float = 0.90, randomize_offers: str = "greedy", concession: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.utility_threshold_ratio = utility_threshold_ratio
        self.randomize_offers = randomize_offers
        self.concession = concession
        self.is_first = None
        self.top_offers = None
        self.best_offer = None
        self.feint_offer = None
        self.endgame_offer = None
        self.instant_walkaway = True
        
        self.best_opponent_offer = None
        self.max_opponent_utility_received = float("-infinity")
        
        self.safe_outcomes = None
        self.om_scored_outcomes = None
        
        # Tracks the last successful concession offer for empty buckets
        self.previous_concession_offer = None

    def _evaluate_om_scores(self):
        """
        Evaluates and caches our utility and the OM's estimated opponent utility 
        for all safe outcomes. Called exactly once at 90% of the game.
        """
        self.om_scored_outcomes = []
        for o in self.safe_outcomes:
            my_u = float(self.negotiator.ufun(o))
            opp_u = 0.0
            
            # Try to get the opponent's estimated utility for this specific deal
            if "opponent_ufun" in self.negotiator.private_info:
                opp_u = float(self.negotiator.private_info["opponent_ufun"](o))
            elif hasattr(self.negotiator, "model") and self.negotiator.model:
                try:
                    opp_u = float(self.negotiator.model(o))
                except:
                    pass
                    
            self.om_scored_outcomes.append((o, my_u, opp_u))

    def _lazy_init(self, state):
        """
        Runs only once at the start of the negotiation. It figures out turn order, 
        filters out suicidal deals, and prepares our standard offers (feint, top deals).
        """
        # Determine if we go first or second
        if self.is_first is None:
            self.is_first = state.current_offer is None

        # Build our deal pools if they haven't been built yet
        if self.top_offers is None and self.negotiator.ufun and self.negotiator.nmi:
            reserved_value = float(self.negotiator.ufun.reserved_value)
            
            outcomes = list(self.negotiator.nmi.outcome_space.enumerate_or_sample())
            sorted_outcomes = sorted(outcomes, key=lambda o: float(self.negotiator.ufun(o)), reverse=True)

            # Walkaway Check: If the best possible deal isn't better than our reservation value, we should just walk away immediately.
            max_possible_utility = float(self.negotiator.ufun(sorted_outcomes[0]))
            if max_possible_utility <= reserved_value:
                self.instant_walkaway = True
                return
            else:
                self.instant_walkaway = False
            
            # Filter out any deals that are at or below our reservation value, as those are effectively "suicidal" offers we would never want to make.
            self.safe_outcomes = [o for o in sorted_outcomes if float(self.negotiator.ufun(o)) > reserved_value]
            if not self.safe_outcomes:
                self.safe_outcomes = [sorted_outcomes[0]]
            
            # Prepare our key offers and cache them for quick access during the negotiation
            self.best_offer = self.safe_outcomes[0]
            max_utility = float(self.negotiator.ufun(self.best_offer))
            
            # Calculate the exact midpoint between the best possible deal and our walkaway value
            target_utility = reserved_value + (max_utility - reserved_value) * 0.5
            
            # Default fallback in case no deals match
            self.feint_offer = self.safe_outcomes[-1]
            
            # Since safe_outcomes is sorted highest-to-lowest, the first deal that 
            # drops strictly below our target is mathematically the closest one
            for o in self.safe_outcomes:
                if float(self.negotiator.ufun(o)) < target_utility:
                    self.feint_offer = o
                    break
            
            target_utility_floor = max(max_utility * self.utility_threshold_ratio, reserved_value)
            self.top_offers = [o for o in self.safe_outcomes if float(self.negotiator.ufun(o)) >= target_utility_floor]
            
            # Cache the endgame offers based on the utility threshold ratio, ensuring we have a fallback if the opponent tries to trap us at the end.
            self.endgame_offer = self.safe_outcomes[0] 
            for o in self.safe_outcomes:
                if float(self.negotiator.ufun(o)) < max_utility:
                    self.endgame_offer = o
                    break

    def __call__(self, state, dest=None):
        self._lazy_init(state)
        
        # Failsafes
        if not self.negotiator.ufun or not self.negotiator.nmi:
            return None
        if self.instant_walkaway:
            return None
            
        # Track the best opponent offer we've seen so far, based on our utility function. 
        # This allows us to have a strong fallback if we need to accept at the end or if we encounter empty buckets during concession.
        if state.current_offer is not None:
            opp_u = float(self.negotiator.ufun(state.current_offer))
            if opp_u > self.max_opponent_utility_received:
                self.max_opponent_utility_received = opp_u
                self.best_opponent_offer = state.current_offer

        total_steps = self.negotiator.nmi.n_steps
        reserved_value = float(self.negotiator.ufun.reserved_value)

        # Determine if we are in the concession phase based on time or steps, 
        # and calculate concession progress as a percentage of the concession phase completed. 
        # This will be used to determine which bucket of offers to consider.
        is_concession_phase = False
        concession_progress = 0.0
        
        if self.concession:
            if total_steps and total_steps > 0:
                concession_start_step = int(total_steps * 0.90)
                if state.step >= concession_start_step:
                    is_concession_phase = True
                    concession_progress = (state.step - concession_start_step) / max(1, (total_steps - concession_start_step))
            else:
                if state.relative_time >= 0.90:
                    is_concession_phase = True
                    concession_progress = (state.relative_time - 0.90) / 0.10

        # Concession Phase Logic: If we're in the concession phase, we want to map our concession progress 
        # into discrete buckets that determine how much we're willing to concede.
        if is_concession_phase:
            if self.om_scored_outcomes is None:
                self._evaluate_om_scores()
                
            # Map progress (0.0 to 1.0) into 10 discrete buckets (0 to 9)
            bucket = int(concession_progress * 10)
            if bucket > 9: 
                bucket = 9
                
            # Calculate strict upper and lower bounds for the current bucket
            upper_pct = 1.0 - (bucket * 0.10)
            lower_pct = max(0.0, 0.90 - (bucket * 0.10))
            
            max_u = float(self.negotiator.ufun(self.best_offer))
            upper_u = reserved_value + (upper_pct * (max_u - reserved_value))
            lower_u = reserved_value + (lower_pct * (max_u - reserved_value))
            
            # Filter deals to ONLY those strictly within the current bucket boundaries
            valid_deals = [item for item in self.om_scored_outcomes if lower_u <= item[1] <= upper_u]
            
            # The Empty Bucket Fallback
            if not valid_deals:
                if self.previous_concession_offer is not None:
                    return self.previous_concession_offer
                return self.best_offer
                
            # Pick the deal that maximizes the opponent's utility. 
            best_deal_tuple = max(valid_deals, key=lambda x: (x[2], x[1]))
            
            # Save this deal so we can fall back to it if the next bucket is empty
            self.previous_concession_offer = best_deal_tuple[0]
            
            return best_deal_tuple[0]

        # Old engame logic: If we're not in the concession phase, we want to ensure that we don't get trapped by the opponent at the end.
        if not is_concession_phase:
            if total_steps and total_steps > 0:
                if self.is_first is False and state.step >= total_steps - 2:
                    return self.endgame_offer
                elif self.is_first is True and state.step >= total_steps - 2:
                    if self.best_opponent_offer is not None and self.max_opponent_utility_received > reserved_value:
                        return self.best_opponent_offer
            else:
                if self.is_first is False and state.relative_time >= 0.98:
                    return self.endgame_offer
                elif self.is_first is True and state.relative_time >= 0.98:
                    if self.best_opponent_offer is not None and self.max_opponent_utility_received > reserved_value:
                        return self.best_opponent_offer
            
        # Offer Selection Logic: If we're not in the concession phase, we select offers based on the specified randomization mode.
        if self.randomize_offers == "feint":
            is_feint_phase = False
            if total_steps and total_steps > 0:
                if state.step < int(total_steps * 0.90):
                    is_feint_phase = True
            else:
                if state.relative_time < 0.90:
                    is_feint_phase = True
                    
            if is_feint_phase:
                return self.feint_offer
            else:
                return self.best_offer
                
        elif self.randomize_offers == "randomize":
            return random.choice(self.top_offers)
            
        else:
            return self.best_offer


class AcceptanceStrategy(ACNext):
    """
    Accepts any offer that yields >= 75% of our Maximum Utility.
    If 'concession' is True, shadows the Bidding Strategy's exact utility target 
    in the last 10% of the game (ACNext logic).
    """
    def __init__(self, offering_strategy, utility_threshold_ratio: float = 0.75, concession: bool = True, *args, **kwargs):
        super().__init__(offering_strategy, *args, **kwargs)
        self.utility_threshold_ratio = utility_threshold_ratio
        self.concession = concession
        self.dynamic_threshold = None

    def __call__(self, state, offer, source=None):
        if offer is None or self.negotiator.ufun is None or not self.negotiator.nmi:
            return ResponseType.REJECT_OFFER
            
        our_utility = float(self.negotiator.ufun(offer))
        reserved_value = float(self.negotiator.ufun.reserved_value)
        max_possible_utility = float(self.negotiator.ufun.max())

        # Immediate Walkaway Check: If the best possible deal isn't better than our reservation value, we should just walk away immediately.
        if max_possible_utility <= reserved_value:
            return ResponseType.END_NEGOTIATION
        
        # Safety Net Check: If the offer is at or below our reservation value, we should reject it immediately, as accepting it would be worse than walking away.
        if our_utility <= reserved_value:
            return ResponseType.REJECT_OFFER
        
        # Calculate our dynamic acceptance threshold based on the maximum possible utility and the specified ratio.
        if self.dynamic_threshold is None:
            self.dynamic_threshold = reserved_value + (max_possible_utility - reserved_value) * self.utility_threshold_ratio

        total_steps = self.negotiator.nmi.n_steps

        # Endgame Acceptance Logic: If we're at the end of the game, we want to be more willing to accept offers that meet 
        # or exceed our dynamic threshold, as we may not have time to make a better offer and we want to avoid getting trapped by the opponent's endgame strategy.
        if total_steps and total_steps > 0:
            if state.step >= total_steps - 1:
                return ResponseType.ACCEPT_OFFER
        else:
            if state.relative_time >= 0.99:
                return ResponseType.ACCEPT_OFFER
                
        # Concession Phase Acceptance Logic: If we're in the concession phase, we want to mirror the offering strategy's concessions 
        # by accepting any offer that meets or exceeds the next offer they would make. This ensures that we don't get out of sync with our own concessions 
        # and accidentally reject offers we would have made ourselves.
        is_concession_phase = False
        if self.concession:
            if total_steps and total_steps > 0:
                if state.step >= int(total_steps * 0.90):
                    is_concession_phase = True
            else:
                if state.relative_time >= 0.90:
                    is_concession_phase = True
                    
        if is_concession_phase:
            next_offer = self.offering_strategy(state)
            if next_offer is not None:
                next_utility = float(self.negotiator.ufun(next_offer))
                if our_utility >= next_utility:
                    return ResponseType.ACCEPT_OFFER

        # Standard Acceptance Logic: If the offer meets or exceeds our dynamic threshold, we should accept it, 
        # as it provides us with a good deal relative to what we could potentially achieve.  
        if our_utility >= self.dynamic_threshold:
            return ResponseType.ACCEPT_OFFER
            
        return ResponseType.REJECT_OFFER


class NashtyNegotiator(BOANegotiator):
    """
    An upgraded Hardliner that leverages exact knowledge of turn order, 
    relative time, Pareto-concessions, and mathematical safety nets.
    """
    def __init__(self, randomize_offers: str = "feint", concession: bool = True, *args, **kwargs):
        
        my_offering = BiddingStrategy(
            utility_threshold_ratio=0.90, 
            randomize_offers=randomize_offers, 
            concession=concession
        )
        
        my_acceptance = AcceptanceStrategy(
            my_offering, 
            utility_threshold_ratio=0.75, 
            concession=concession
        )
        
        kwargs |= dict(
            offering=my_offering,
            acceptance=my_acceptance,
            model=GSmithFrequencyModel(),
        )
        super().__init__(*args, **kwargs)