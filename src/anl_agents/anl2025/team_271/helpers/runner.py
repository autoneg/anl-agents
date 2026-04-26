"""
A helper function to run a tournament with your agent.

You only need to change the name of the class implementing your agent at the top of this file.
"""

import time
import random

from anl2025 import (
    anl2025_tournament,
    DEFAULT_TOURNAMENT_PATH,
    DEFAULT_ANL2025_COMPETITORS,
    MultidealScenario,
    make_multideal_scenario,
    make_dinners_scenario,
    make_job_hunt_scenario,
    make_target_quantity_scenario,
    run_session,
    CenterUFun
)


def run_a_tournament(
    TestedNegotiators,
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


    from anl2025.negotiator import Conceder2025
    from negmas.helpers import humanize_time, unique_name
    from rich import print

    start = time.perf_counter()
    # name = (
    #     unique_name(f"test{TestedNegotiators().type_name.split('.')[-1]}", sep="")
    #     if not nologs
    #     else None
    # )


    scenarios = [make_multideal_scenario(nedges=3, nissues=2, nvalues=2) for _ in range(1)]
    scenariosbig = [make_multideal_scenario(nedges=4, ) for _ in range(2)]+[make_dinners_scenario(n_friends=4, n_days=5) for _ in range(2)]+[make_job_hunt_scenario(n_employers=4) for _ in range(2)]+[make_target_quantity_scenario(n_suppliers=4) for _ in range(2)]
    if small:
        results = anl2025_tournament(
            competitors=tuple(TestedNegotiators),
            scenarios=scenarios,
            # non_comptitor_types=tuple([Conceder2025]),
            n_repetitions=1,
            n_jobs=-1 if debug else 0,
            verbose=False,
        )
    else:
        results = anl2025_tournament(
            competitors=tuple(list(set(list(DEFAULT_ANL2025_COMPETITORS) + TestedNegotiators))),
            scenarios=scenariosbig,
            # non_comptitor_types=tuple(list(DEFAULT_ANL2025_COMPETITORS)),
            n_repetitions=n_repetitions,
            n_jobs=-1 if debug else 0,
            verbose=True,
        )
    pass # print("Final Scores:")
    pass # print(results.final_scores)
    pass # print("Weighted Average:")
    pass # print(results.weighted_average)
    pass # print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    # if name is not None:
    #     print(f"You can see all logs at {DEFAULT_TOURNAMENT_PATH / name}")


def run_a_negotiation(TestedNegotiator):
    centeragent = TestedNegotiator
    edgeagents = tuple(list(set(list(DEFAULT_ANL2025_COMPETITORS))))
    scenario = make_multideal_scenario(nedges=3, nissues=2, nvalues=2, center_reserved_value_min=0.1, center_reserved_value_max=0.8)
    pass # print(scenario)

    results = run_session(
        scenario=scenario,
        center_type=centeragent,
        edge_types=edgeagents,  # type: ignore
        nsteps=10,
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
    pass # print(f"Agreement: {results.agreements}")

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
    pass # print(f"Center Utility: {results.center_utility}")


def make_random_dinners_scenario() -> MultidealScenario:
    n_friends = random.randint(2, 5)
    n_days = random.randint(n_friends, 7)
    center_reserved_value = (0.0, 0.0)
    edge_reserved_values = (0.0, 0.4)
    return make_dinners_scenario(
        n_friends=n_friends,
        n_days=n_days,
        center_reserved_value=center_reserved_value,
        edge_reserved_values=edge_reserved_values)

def make_random_job_hunt_scenario() -> MultidealScenario:
    n_employers = random.randint(2, 5)
    work_days = random.randint(5, 7)
    return make_job_hunt_scenario(
        n_employers=n_employers,
        work_days=work_days,
    )

def make_random_target_quantity_scenario() -> MultidealScenario:
    n_suppliers = random.randint(2, 5)
    quantity = random.randint(4,6)
    collector_reserved_value = random.uniform(0.0, 0.0)
    supplier_reserved_values = random.uniform(0.0, 0.4)
    return make_target_quantity_scenario(
        n_suppliers=n_suppliers,
        quantity=quantity,
        collector_reserved_value=collector_reserved_value,
        supplier_reserved_values=supplier_reserved_values,
    )

def make_random_multi_deal_scenario() -> MultidealScenario:
    nedges = random.randint(2, 5)
    nissues = random.randint(2, 4)
    nvalues = random.randint(4,7)
    center_reserved_value_min = 0.0
    center_reserved_value_max = 0.0
    edge_reserved_value_min = 0.1
    edge_reserved_value_max = 0.4

    return make_multideal_scenario(
        nedges=nedges,
        nissues=nissues,
        nvalues=nvalues,
        center_reserved_value_min=center_reserved_value_min,
        center_reserved_value_max=center_reserved_value_max,
        edge_reserved_value_min=edge_reserved_value_min,
        edge_reserved_value_max=edge_reserved_value_max,
    )

def agent_tester(TestedNegotiators, n_repetitions=1, debug=False):

    from anl2025.negotiator import Conceder2025
    from negmas.helpers import humanize_time
    from rich import print

    start = time.perf_counter()

    results = anl2025_tournament(
        competitors=tuple(list(set(list(DEFAULT_ANL2025_COMPETITORS) + TestedNegotiators))),
        scenarios=tuple(
            [make_random_dinners_scenario() for _ in range(50)]
            + [make_random_job_hunt_scenario() for _ in range(50)]
            + [make_random_target_quantity_scenario() for _ in range(50)]
            + [make_random_multi_deal_scenario() for _ in range(50)]
        ),
        n_repetitions=n_repetitions,
        n_jobs=-1 if debug else 0,
        verbose=True,
    )
    pass # print("Final Scores:")
    pass # print(results.final_scores)
    pass # print("Weighted Average:")
    pass # print(results.weighted_average)
    pass # print(f"Finished in {humanize_time(time.perf_counter() - start)}")