"""
This is the code that is part of Tutorial 1 for the ANL 2025 competition, see URL.

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""
import pathlib

from anl2025.scenario import MultidealScenario
from anl2025 import (
    run_session,
    make_job_hunt_scenario,
    make_target_quantity_scenario,
    load_example_scenario,
)
from anl2025.tournament import anl2025_tournament
from anl2025.ufun import CenterUFun
from anl2025.negotiator import Boulware2025, Random2025, Linear2025, Conceder2025, Random2025
from anl2025.scenario import make_multideal_scenario

import numpy as np
import matplotlib.pyplot as plt

import random

def run_for_debug(
    TestedNegotiator,
    n_repetitions=5,
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

    from anl2025 import (
        anl2025_tournament,
        DEFAULT_TOURNAMENT_PATH,
        DEFAULT_ANL2025_COMPETITORS,
        make_multideal_scenario,
    )


    from anl2025.negotiator import Conceder2025
    from negmas.helpers import humanize_time, unique_name
    from rich import print

    start = time.perf_counter()
    name = (
        unique_name(f"test{TestedNegotiator().type_name.split('.')[-1]}", sep="")
        if not nologs
        else None
    )

    scenarios = [make_multideal_scenario(nedges=3, nissues=2, nvalues=2) for _ in range(1)]
    scenariosbig = [make_multideal_scenario(nedges=3, ) for _ in range(2)]
    if small:
        anl2025_tournament(
            competitors=tuple([TestedNegotiator]),
            scenarios=scenarios,
            non_comptitor_types=tuple([Conceder2025]),
            n_repetitions=1,
            n_jobs=-1 if debug else 0,
            verbose=False,
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
    pass # print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    if name is not None:
        pass # print(f"You can see all logs at {DEFAULT_TOURNAMENT_PATH / name}")

def run_negotiation(centeragent):
    # agents:
    edgeagents = [
        Boulware2025,
        Linear2025,
        Conceder2025,
        # Random2025,
        # Shochan2025, 
        # AgentRenting2025,
    ]

    # scenarios = [
    #     MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/dinners")),
    # ]

    scenarios = [
        MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/TargetQuantity_example")),
    ]


    # scenarios = [
    #     MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/job_hunt_target")),
    # ]


    results = run_session(
        scenario=scenarios[0],
        center_type=centeragent,
        edge_types=edgeagents,  # type: ignore
        nsteps=100,
        #  verbose=verbose,
        #  keep_order=keep_order,
        #  share_ufuns=share_ufuns,
        #  atomic=atomic,
        #  output=output,
        #  dry=dry,
        #  method=DEFAULT_METHOD,
        #  sample_edges=False,
    )

    # print some results
    pass # print(f"Center utility: {results.center_utility}")
    pass # print(f"Edge Utilities: {results.edge_utilities}")
    # print(f"Score: {(results.center_utility + np.mean(results.edge_utilities)) / 2}")
    # print(f"Agreement: {results.agreements}")

    # extra: for nicer lay-outing and more results:
    cfun = results.center.ufun

    assert isinstance(cfun, CenterUFun)
    side_ufuns = cfun.side_ufuns()

    for i, (e, m, u) in enumerate(
        zip(results.edges, results.mechanisms, side_ufuns, strict=True)  # type: ignore
    ):
        print(
            f"{i:02}: Mechanism {m.name} between ({m.negotiator_ids}) ended in {m.current_step} ({m.relative_time:4.3}) with {m.agreement}: "
            f"Edge Utility = {e.ufun(m.agreement) if e.ufun else 'unknown'}, "
            f"Side Utility = {u(m.agreement) if u else 'unknown'}"
        )
        for outcome in m.outcome_space.enumerate_or_sample():
            pass # print(f"Outcome: {outcome} SUtility: {u(outcome)}")
    # print(f"Center Utility: {results.center_utility}")

    return results

def run_tournament(myagent):
    # scenarios = [
    #     MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/dinners")),
    #     MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/TargetQuantity_example")),
    #     MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/job_hunt_target")),
    # ]

    scenarios = [
        MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/TargetQuantity_example")),
        MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/job_hunt_target")),
        MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/dinners")),
    ]


    competitors = (
        myagent,
        Boulware2025,
        Linear2025,
        Conceder2025,
        Random2025,
    )


    results = anl2025_tournament(
        scenarios=scenarios,
        n_jobs=-1,
        n_steps=50,
        # n_steps=random.randint(10, 1000),
        n_repetitions = 1,
        competitors=competitors,
        verbose=False,
        #  no_double_scores=False,
    )
    pass # print({k:f'{v:.3f}' for k,v in sorted(results.final_scores.items(), key=lambda x: x[1], reverse=True)})
    pass # print({k:f'{v:.3f}' for k,v in sorted(results.weighted_average.items(), key=lambda x: x[1], reverse=True)})

    return results

def visualize(results):
    for _, m in enumerate(results.mechanisms):
        plot_result(m)

def plot_result(m):
    m.plot(save_fig=False)
    plt.show()
    plt.close()

# def run_generated_negotiation():
#     scenario = make_multideal_scenario(nedges=8)
#     scenario = make_job_hunt_scenario()
#     scenario = make_target_quantity_scenario()
#     results = run_session(scenario)
#     print(f"Center utility: {results.center_utility}")
#     print(f"Edge Utilities: {results.edge_utilities}")