"""Tests for the ANL 2026 participants.

Mirrors test_agents2025.py (registry counts) and test_tournament.py
(actually running each agent). The full participant set is materialised
from the local DB + restores by scmlweb/python/update_agents_repo.py;
winners are populated once announced.
"""

import warnings

from pytest import mark

from anl.anl2024.runner import anl2024_tournament
from anl_agents import get_agents

warnings.filterwarnings("ignore")

STEPS = 5
OUTCOMES = 9

# Resolve once at collection time (the year package imports defensively,
# so a single broken submission cannot break collection of the rest).
QUALIFIED_2026 = get_agents(2026, as_class=True, qualified_only=True)


def test_get_agents_2026_all():
    agents = get_agents(2026, as_class=False)
    assert len(agents) == 39


def test_get_agents_2026_qualified_excludes_disqualified():
    # qualified == not disqualified; the two DQ'd 2026 agents have no
    # local code so are absent entirely, hence all == qualified for now.
    assert len(get_agents(2026, qualified_only=True)) == 37


def test_get_agents_2026_finalists():
    assert len(get_agents(2026, finalists_only=True)) == 17


def test_get_agents_2026_winners_pending():
    # Populated by set_winners.py once announced.
    assert len(get_agents(2026, winners_only=True)) == 0


@mark.parametrize("agent", QUALIFIED_2026, ids=[a.__name__ for a in QUALIFIED_2026])
def test_can_run_2026(agent):
    """Each agent must complete a small ANL-2024-style tournament."""
    anl2024_tournament(
        n_scenarios=1,
        n_steps=STEPS,
        n_outcomes=OUTCOMES,
        competitors=[agent],
        nologs=True,
        verbosity=0,
    )
