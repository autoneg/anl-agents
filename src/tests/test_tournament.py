from pytest import mark
from itertools import product

from anl_agents import get_agents
from anl.anl2024.runner import anl2024_tournament

tracks = ["advantage"]
years = [2024]
STEPS = 5
OUTCOMES = 9


@mark.parametrize(["version", "track"], product(years, tracks))
def test_can_run_winners(version, track):
    anl2024_tournament(
        n_scenarios=2,
        n_steps=STEPS,
        n_outcomes=OUTCOMES,
        competitors=get_agents(version, track=track, winners_only=True),
        nologs=True,
        verbosity=0,
    )


# @mark.parametrize(["version", "track"], product(years, tracks))
# def test_can_run_qualified(version, track):
#     anl2024_tournament(
#         n_scenarios=2,
#         n_steps=STEPS,
#         n_outcomes=OUTCOMES,
#         competitors=get_agents(version, track=track, qualified_only=True),
#     )
#
#
# @mark.parametrize(["version", "track"], product(years, tracks))
# def test_can_run_finalists(version, track):
#     anl2024_tournament(
#         n_scenarios=2,
#         n_steps=STEPS,
#         n_outcomes=OUTCOMES,
#         competitors=get_agents(version, track=track, finalists_only=True),
#     )
#
#
# @mark.parametrize(["version", "track"], product(years, tracks))
# def test_can_run(version, track):
#     anl2024_tournament(
#         n_scenarios=2,
#         n_steps=STEPS,
#         n_outcomes=OUTCOMES,
#         competitors=get_agents(version, track=track),
#     )
