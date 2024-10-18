"""
A helper function to run a tournament with your agent.

You only need to change the name of the class implementing your agent at the top of this file.
"""


def run_a_tournament(
    TestedNegotiator,
    n_repetitions=4,
    n_outcomes=1000,
    n_scenarios=12,
    debug=False,
    nologs=False,
    small=False,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
       TestedNegotiator: Negotiator type to be tested
       n_repetitions: The number of repetitions of each scenario tested
       n_outcomes: Number of outcomes in the domain (makes sure this is between 900 and 1100)
       n_scenarios: Number of different scenarios generated
       debug: Pass True here to run the tournament in serial, increase verbosity, and fails on any exception
       nologs: If passed, no logs will be stored
       small: if set to True, the tournament will be very small and run in a few seconds.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_repetitions` value

    """
    import time

    from anl.anl2024 import (
        DEFAULT_AN2024_COMPETITORS,
        DEFAULT_TOURNAMENT_PATH,
        anl2024_tournament,
    )
    from anl.anl2024.negotiators import Conceder, Boulware, MiCRO, RVFitter, NashSeeker, Linear
    from negmas.helpers import humanize_time, unique_name
    from rich import print

    start = time.perf_counter()
    name = (
        unique_name(f"test{TestedNegotiator().type_name.split('.')[-1]}", sep="")
        if not nologs
        else None
    )
    if small:
        anl2024_tournament(
            competitors=tuple([TestedNegotiator, Boulware]),
            #RVFitter   - Ingo:0.48 vs 0.31
            #MiCRO      - Ingo:0.03 vs 0.00
            #Linear     - Ingo:0.37 vs 0.29
            #NashSeeker - Ingo:0.4 vs 0.41
            #Boulware   - Ingo:0.34 vs 0.26
            #Conceder   - Ingo:0.36 vs 0.24
            n_scenarios=4,
            n_outcomes=n_outcomes,
            n_repetitions=4,
            njobs=-1 if debug else 0,
            verbosity=2 if debug else 2,
            plot_fraction=0,
            name=name,
        ).final_scores
    else:
        anl2024_tournament(
            competitors=tuple([TestedNegotiator] + list((Boulware, RVFitter, NashSeeker, Linear))),
            n_scenarios=n_scenarios,
            n_outcomes=n_outcomes,
            n_repetitions=n_repetitions,
            njobs=-1 if debug else 0,
            verbosity=2 if debug else 2,
            plot_fraction=0,
            name=name,
        ).final_scores
    pass # print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    if name is not None:
        pass # print(f"You can see all logs at {DEFAULT_TOURNAMENT_PATH / name}")