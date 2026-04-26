from rich import print

from anl_agents import get_agents


def test_get_agents_per_year_all():
    agents = get_agents(2025, as_class=False)
    assert len(agents) == 37
    print(agents)


def test_get_agents_per_year_qualified():
    agents = get_agents(2025, qualified_only=True, as_class=False)
    assert len(agents) == 17
    print(agents)


def test_get_agents_per_year_finalists():
    agents = get_agents(2025, finalists_only=True, as_class=False)
    assert len(agents) == 12
    print(agents)


def test_get_agents_per_year_winners():
    agents = get_agents(2025, winners_only=True, as_class=False)
    assert len(agents) == 3
    print(agents)
