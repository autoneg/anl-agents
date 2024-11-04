import random


class BiddingStrategy:
    """
    Class responsible for the bidding strategy of our agent
    """

    history: list = []
    "All bids received from the opponent"
    order: list = []
    "Order of strategies our agent will go through"
    order_index: int = 0
    "Index of current strategy"
    phase_settings: dict
    "Settings of the agent"
    current_phase: str
    """One of the possible phases of the agent, which switch depending on how much
    time has elapsed from the start of the session\n
    analysis -> stubborn -> compromising -> conceding"""
    analysis_bids: int = 0
    "How many bids were already placed in analysis phase"

    stubborn_bid_index: int = 0
    compromising_bid_index: int = 0
    reservation_value: int = 0

    def __init__(self, agent, phase_settings=None):
        """
        Initialization

        agent: Our set up agent
        phase_settings: Dictionary with configuration
        """

        self.steps_count = (
            agent.nmi.n_steps
            if agent.nmi.n_steps is not None
            else int(
                (agent.nmi.state.time + 1e-6) / (agent.nmi.state.relative_time + 1e-6)
                + 0.5
            )
        )
        self.opponent_reserved_value = None

        # Set random
        random.seed()

        from .helpers.config import DEFAULT_SETTINGS

        # Parse setting to set the order of the bidding strategy
        self.parse_settings(DEFAULT_SETTINGS)
        self.reservation_value = agent.reserved_value

        # Save all possible outcomes
        outcomes = [
            self.Outcome(
                x,
                agent.ufun(x),
                agent.opponent_ufun(x),
            )
            for x in agent.nmi.outcome_space.enumerate_or_sample()
        ]

        # Sort outcomes based on our utility
        self.sorted_outcomes = sorted(outcomes, key=lambda x: x.utility, reverse=True)

        # Sort outcomes based on combined utility
        self.sorted_combined_outcomes = sorted(
            outcomes, key=lambda x: x.combined_utility + x.utility, reverse=True
        )

        # Sorted outcomes in which our utility is higher
        self.sorted_advantage_outcomes = [
            x for x in self.sorted_outcomes if x.utility >= x.opponent_utility
        ]

        # Sorted utility for 'stubborn' phase
        self.sorted_stubborn_outcomes = sorted(
            [
                x
                for x in self.sorted_combined_outcomes
                if x.utility >= self.reservation_value
            ],
            key=lambda x: x.combined_utility + x.utility,
            reverse=True,
        )

        # All outcomes that are above our reservation value.
        # Might contain offers where the opponent is better of
        self.compromising_outcomes = sorted(
            [x for x in self.sorted_outcomes if x.utility >= self.reservation_value],
            key=lambda x: x.combined_utility
            + x.utility
            * self.phase_settings["compromising"]["compromising_util_multiplier"],
            reverse=True,
        )

        # [x for x in self.sorted_combined_outcomes if x.utility >= self.reservation_value]
        # self.conceding = [x for x in self.sorted_outcomes if x.utility >= self.reserved_value and x.utility >= x.opponent_utility]
        self.stubborn_bid_index = 0
        self.compromising_bid_index = 0

        self.analysis_bids = 0
        self.history = []

    class Outcome:
        """
        Class to store properties of an outcome
        """

        def __init__(self, outcome, utility, opponent_utility):
            self.outcome = outcome
            self.utility = utility
            self.opponent_utility = opponent_utility

            # Combined utility of both agents
            self.combined_utility = utility + opponent_utility

            # Advantage of an agent. Range [-1; 1]
            # difference > 0 - our agent has advantage
            # difference < 0 - opponent has advantage
            self.difference = utility - opponent_utility

    def parse_settings(self, settings):
        """
        Method to extract settings and set up the bidding strategy

        settings: Dictionary with configuration
        """
        # Get order of the phases
        self.order = list(settings["bidding-phases"].keys())
        if self.steps_count < 35:
            self.order = [
                x
                for x in self.order
                if "analysis" not in x
                and "stubborn" not in x
                and "conceding" not in x
                and "irrational" not in x
            ]

        # Set current phase
        self.current_phase = self.order[0]
        # Store settings
        self.phase_settings = settings["bidding-phases"]

    def analysis_strategy(self):
        """
        Method responsible for the analysis phase
        """
        # Keep track of how many bids where done during analysis phase so that we can end it,
        # when we have enough data collected
        self.analysis_bids += 1

        # We don't want the situation where our opponent accepts our analytical bids,
        # thus, we try to keep them as irrational as possible
        #
        # Randomly select index of the analytical bid
        # index = random.randint(0, 3)

        # With 50% chance we select the bid which is either:
        # 1. very disadvantageous to our opponent
        # 2. very disadvantageous to both of us, but we still have a higher utility
        if bool(random.getrandbits(1)):
            # Disadvantageous to our opponent
            return self.sorted_outcomes[0].outcome
        else:
            # Disadvantageous to both of us
            return self.sorted_advantage_outcomes[-1].outcome

    def stubborn_strategy(self):
        """
        Method responsible for the stubborn phase
        """
        if self.phase_settings["stubborn"]["irrationality"] > random.uniform(0.0, 1.0):
            return self.irrational_strategy()

        bid = self.sorted_stubborn_outcomes[self.stubborn_bid_index]
        self.stubborn_bid_index += 1

        if self.stubborn_bid_index >= int(
            len(self.sorted_stubborn_outcomes)
            * self.phase_settings["stubborn"]["stubborn_bid_index_limit"]
        ):
            self.stubborn_bid_index = 0

        return bid.outcome

    def compromising_strategy(self):
        """
        Method responsible for the compromising phase
        """
        bid = self.compromising_outcomes[self.compromising_bid_index]
        if self.compromising_bid_index >= int(
            len(self.compromising_outcomes)
            * self.phase_settings["compromising"]["compromising_bid_index_limit"]
        ):
            self.compromising_bid_index = 0
        else:
            self.compromising_bid_index += 1

        return bid.outcome

    def irrational_strategy(self):
        """
        Method responsible for the irrational phase
        """

        return self.sorted_advantage_outcomes[-1].outcome

    def adapting_strategy(self):
        """
        Method responsible for the conceding phase
        """
        # Sort historical bids based on our utility and the difference between our and opponents utility
        # to maximize overall value of the bid
        if self.opponent_reserved_value and self.opponent_reserved_value <= 0.75:
            self.history = sorted(
                [x for x in self.history if x.utility >= self.reservation_value],
                key=lambda x: x.utility,
                reverse=True,
            )

            if len(self.history) > 0:
                tmp_bids = [
                    x
                    for x in self.sorted_outcomes
                    if (x.utility >= self.reservation_value)
                    and (x.opponent_utility >= self.opponent_reserved_value - 0.03)
                    and (x.utility >= self.history[0].utility)
                ]
            else:
                tmp_bids = [
                    x
                    for x in self.sorted_outcomes
                    if (x.utility >= self.reservation_value)
                    and (x.opponent_utility >= self.opponent_reserved_value - 0.03)
                ]

            bids = sorted(
                tmp_bids, key=lambda x: x.combined_utility + 3 * x.utility, reverse=True
            )

            if len(bids) > 0:
                return bids[0].outcome
            else:
                return self.compromising_strategy()
        else:
            return self.compromising_strategy()

        # return self.history[0].outcome

    def update_phase(self, time_elapsed):
        """
        Phase switching mechanism

        time_elapsed: How much time elapsed from the start of the tournament
        """
        if self.current_phase == "analysis":
            # If we bid required number of bids for analysis, move to the next phase
            if self.analysis_bids == self.phase_settings["analysis"]["analysis_length"]:
                self.order_index += 1
                self.current_phase = self.order[self.order_index]
        # If we reached time limit of the phase, move to the next one
        elif self.phase_settings[self.current_phase]["time_limit"] < time_elapsed:
            if not (self.order_index == len(self.order) - 1):
                self.order_index += 1
            self.current_phase = self.order[self.order_index]

        return self.current_phase

    def get_next_bid(self, state, opponent_reserved_value=None):
        """
        Method responsible for provision of the next bid.

        state: The offer from the opponent (`None` if you are just starting the negotiation)
        """
        self.opponent_reserved_value = opponent_reserved_value
        # Store the offer in history
        self.history.extend(
            [x for x in self.sorted_outcomes if x.outcome == state.current_offer]
        )
        # Update current phase based on the elapsed time
        current_phase = self.update_phase(state.relative_time)
        # Return bid based on the current phase
        return self.__getattribute__(f"{current_phase.split('-')[0]}_strategy")()
