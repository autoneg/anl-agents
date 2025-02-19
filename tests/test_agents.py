from itertools import product

from pytest import mark
from rich import print

from anl_agents import get_agents

tracks = ["advantage", "utility", "nash", "kalai", "kalai-smorodonisky", "welfare"]
years = [2024]


@mark.parametrize(["version", "track"], product(years, tracks))
def test_get_agents_per_year_all(version, track):
    agents = get_agents(version, track=track, as_class=False)
    assert len(agents) == 26
    print(agents)


@mark.parametrize(["version", "track"], product(years, tracks))
def test_get_agents_per_year_qualified(version, track):
    agents = get_agents(version, track=track, qualified_only=True, as_class=False)
    assert len(agents) == 22
    print(agents)


@mark.parametrize(["version", "track"], product(years, tracks))
def test_get_agents_per_year_finalists(version, track):
    agents = get_agents(version, track=track, finalists_only=True, as_class=False)
    assert len(agents) == 10
    print(agents)


def test_get_agents_per_year_winners_advantage():
    agents = get_agents(2024, track="advantage", winners_only=True, as_class=False)
    assert len(agents) == 3
    print(agents)


def test_get_agents_per_year_winners_nash():
    agents = get_agents(2024, track="nash", winners_only=True, as_class=False)
    assert len(agents) == 1
    print(agents)
