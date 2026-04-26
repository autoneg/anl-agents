# Settings which define how agent will be running
DEFAULT_SETTINGS = {
    # Phases our agent will go through, should be added in order in which they
    # should be switched
    "bidding-phases": {
        "analysis": {
            # Length in bids of the analysis phase
            "analysis_length": 20,
        },
        "stubborn": {
            # Indicates when phase should be switched to the next one
            "time_limit": 0.75,
            "stubborn_bid_index_limit": 0.1,
            "irrationality": 0.01,
        },
        "compromising": {
            "time_limit": 0.94,
            "compromising_bid_index_limit": 0.1,
            "compromising_util_multiplier": 0.8,
        },
        "adapting": {
            "time_limit": 1,
        },
    },
    "acceptance": {"k": 0.1, "e": 1, "time_threshold": 0.94},
    # Settings to define how opponent model works
    # rv_feature_mode:
    # 0 : use only round # for prediction
    # 1 : use round # and NE distance for prediction
    #
    # step_class:
    # 0 : 0 - 100 step size
    # 1 : 100 - 200
    # 2 : 200 - 500
    # 3 : 500 - 1000
    # 4 : 1000 - 2500
    # 5 : 2500 - 5000
    # 6 : 5000 - 10000
    #
    # phases:
    # 0 : 0/8 - 4/8
    # 1 : 4/8 - 6/8
    # 2 : 6/8 - 7/8
    # 3 : 7/8 - 8/8 of the session.
    #
    # possible window_sizes : 3, 5, 10, 25, 50, 100, 200
    # window_size should be at least (step_size / 2)
    "opponent_model": {
        "rv_feature_mode": 1,  # if we use opponents distance to Nash utility in the RV prediction
        "step_class": {
            "step_0": {
                "phase_0": 50,  # if the session is 0-100 rounds and we're at the first 4/8 of it --> window_size = 3
                "phase_1": 50,  # if the session is 0-100 rounds and we're at 4/8 - 6/8 of it --> window_size = 5
                "phase_2": 50,
                "phase_3": 25,
            },
            "step_1": {"phase_0": 50, "phase_1": 25, "phase_2": 5, "phase_3": 5},
            "step_2": {"phase_0": 10, "phase_1": 5, "phase_2": 5, "phase_3": 50},
            "step_3": {"phase_0": 5, "phase_1": 5, "phase_2": 500, "phase_3": 200},
            "step_4": {"phase_0": 5, "phase_1": 25, "phase_2": 25, "phase_3": 3},
            "step_5": {"phase_0": 50, "phase_1": 500, "phase_2": 25, "phase_3": 5},
            "step_6": {"phase_0": 5, "phase_1": 200, "phase_2": 50, "phase_3": 50},
        },
    },
}
