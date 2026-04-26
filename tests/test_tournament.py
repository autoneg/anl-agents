from itertools import product

from anl.anl2024.runner import anl2024_tournament
from anl2025.common import RunParams
from anl2025.tournament import Tournament
from pytest import mark

from anl_agents import get_agents

tracks = ["advantage"]
STEPS = 5
OUTCOMES = 9


@mark.parametrize(["version", "track"], product([2024], tracks))
def test_can_run_winners2024(version, track):
    anl2024_tournament(
        n_scenarios=2,
        n_steps=STEPS,
        n_outcomes=OUTCOMES,
        competitors=get_agents(version, track=track, winners_only=True),
        nologs=True,
        verbosity=0,
    )


@mark.parametrize(["version", "track"], product([2025], tracks))
def test_can_run_winners2025(version, track):
    Tournament.from_scenarios(
        competitors=get_agents(2025, winners_only=True),
        run_params=RunParams(),
        n_generated=2,
    ).run(n_repetitions=0, path=None)


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
