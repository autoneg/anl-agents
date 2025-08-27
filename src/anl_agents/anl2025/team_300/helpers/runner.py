

def run_a_tournament(
    TestedNegotiator,
    n_repetitions=5,
    debug=False,
    nologs=False,
    small=False,
):

    import time

    from anl2025 import (
        anl2025_tournament,
        DEFAULT_TOURNAMENT_PATH,
        DEFAULT_ANL2025_COMPETITORS,
        make_multideal_scenario,
    )

    from anl2025.negotiator import Conceder2025,Boulware2025,Linear2025,Random2025,TimeBased2025
    # from anl2025.negotiator import ANL2024_AVAILABLE
    #
    from negmas.helpers import humanize_time, unique_name
    # from rich import print


    start = time.perf_counter()


    name = (
        unique_name(f"test{TestedNegotiator().type_name.split('.')[-1]}", sep="")
        if not nologs
        else None
    )


    scenarios = [make_multideal_scenario(nedges=3, nissues=3, nvalues=3) for _ in range(1)]


    scenariosbig = [make_multideal_scenario(nedges=3) for _ in range(2)]


    if small:

        anl2025_tournament(
            competitors=tuple([TestedNegotiator]),
            scenarios=scenarios,
            non_comptitor_types=tuple([TestedNegotiator]),
            # non_comptitor_types=tuple([Boulware2025,Linear2025,Conceder2025,Random2025,TimeBased2025,TestedNegotiator]),


            n_repetitions=1,
            n_jobs=-1 if debug else 0,
            verbose=True,
        ).final_scores
    else:

        anl2025_tournament(
            competitors=tuple([TestedNegotiator]),
            scenarios=scenariosbig,
            non_comptitor_types=tuple(list(DEFAULT_ANL2025_COMPETITORS)),
            n_repetitions=n_repetitions,
            n_jobs=-1 if debug else 0,
            verbose=True,
        ).final_scores


    pass # print(f"Completion time: {humanize_time(time.perf_counter() - start)}")


    if name is not None:


        pass # print(f"You can view all logs in the following path: {DEFAULT_TOURNAMENT_PATH / name}")